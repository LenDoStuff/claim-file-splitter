from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


DocumentType = Literal[
    "repair_invoices",
    "appraisals",
    "communications",
    "police_reports",
    "photos",
    "payments",
    "medical",
    "legal_correspondence",
    "other",
]

DOCUMENT_TYPES: tuple[DocumentType, ...] = (
    "repair_invoices",
    "appraisals",
    "communications",
    "police_reports",
    "photos",
    "payments",
    "medical",
    "legal_correspondence",
    "other",
)

DOCUMENT_TYPE_PREFIXES: dict[DocumentType, str] = {
    "repair_invoices": "repair_invoice",
    "appraisals": "appraisal",
    "communications": "communication",
    "police_reports": "police_report",
    "photos": "photo_section",
    "payments": "payment_document",
    "medical": "medical_document",
    "legal_correspondence": "legal_correspondence",
    "other": "document",
}


@dataclass(frozen=True)
class PageImage:
    page_number: int
    mime_type: str
    width_px: int
    height_px: int
    byte_size: int
    data_uri: str = field(repr=False)
    path: Path | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "page": self.page_number,
            "mime_type": self.mime_type,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "byte_size": self.byte_size,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = self.to_prompt_dict()
        if self.path is not None:
            payload["path"] = str(self.path)
        return payload


def normalize_document_type(value: str | None) -> DocumentType:
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
        "legal_correspondence": "legal_correspondence",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in DOCUMENT_TYPES:
        return normalized  # type: ignore[return-value]
    return "other"


@dataclass(frozen=True)
class PageFeatures:
    source_path: Path
    page_number: int
    text: str
    word_count: int
    char_count: int
    image_count: int
    is_image_only: bool
    may_require_ocr: bool
    image: PageImage | None = None

    def to_prompt_dict(self, max_text_chars: int) -> dict[str, Any]:
        text = self.text.strip()
        if len(text) > max_text_chars:
            text = text[:max_text_chars].rstrip() + "\n[truncated]"
        return {
            "page": self.page_number,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "image_count": self.image_count,
            "is_image_only": self.is_image_only,
            "may_require_ocr": self.may_require_ocr,
            "text_excerpt": text,
        }

    def to_image_prompt_dict(self) -> dict[str, Any]:
        return {
            "page": self.page_number,
            "rendered_image": self.image.to_prompt_dict() if self.image else None,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "page": self.page_number,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "image_count": self.image_count,
            "is_image_only": self.is_image_only,
            "may_require_ocr": self.may_require_ocr,
        }
        if self.image is not None:
            payload["rendered_image"] = self.image.to_manifest_dict()
        return payload


@dataclass(frozen=True)
class PageDecision:
    page_number: int
    document_type: DocumentType
    starts_new_document: bool
    title: str = ""
    confidence: float = 0.0
    reason: str = ""

    def normalized(self) -> "PageDecision":
        return PageDecision(
            page_number=self.page_number,
            document_type=normalize_document_type(self.document_type),
            starts_new_document=bool(self.starts_new_document),
            title=(self.title or "").strip(),
            confidence=max(0.0, min(1.0, float(self.confidence or 0.0))),
            reason=(self.reason or "").strip(),
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentSegment:
    segment_id: int
    document_type: DocumentType
    start_page: int
    end_page: int
    title: str
    confidence: float
    reasons: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return self.end_page - self.start_page + 1

    def to_manifest_dict(self, output_path: Path | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "segment_id": self.segment_id,
            "document_type": self.document_type,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_count": self.page_count,
            "title": self.title,
            "confidence": round(self.confidence, 4),
            "reasons": self.reasons,
        }
        if output_path is not None:
            payload["output_path"] = str(output_path)
        return payload


@dataclass(frozen=True)
class WrittenDocument:
    segment: DocumentSegment
    output_path: Path

    def to_manifest_dict(self) -> dict[str, Any]:
        return self.segment.to_manifest_dict(self.output_path)


@dataclass(frozen=True)
class ClassificationBatch:
    batch_number: int
    start_page: int
    end_page: int
    page_numbers: list[int]
    rolling_context: dict[str, Any]
    reconciliation_messages: list[str] = field(default_factory=list)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "batch_number": self.batch_number,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "page_numbers": self.page_numbers,
            "rolling_context": self.rolling_context,
            "reconciliation_messages": self.reconciliation_messages,
        }


@dataclass(frozen=True)
class SplitResult:
    source_pdf: Path
    output_dir: Path
    pages: list[PageFeatures]
    page_decisions: list[PageDecision]
    segments: list[DocumentSegment]
    written_documents: list[WrittenDocument]
    manifest_path: Path
    classification_batches: list[ClassificationBatch] = field(default_factory=list)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "source_pdf": str(self.source_pdf),
            "output_dir": str(self.output_dir),
            "page_count": len(self.pages),
            "document_count": len(self.segments),
            "pages": [page.to_manifest_dict() for page in self.pages],
            "page_decisions": [
                decision.to_manifest_dict() for decision in self.page_decisions
            ],
            "classification_batches": [
                batch.to_manifest_dict() for batch in self.classification_batches
            ],
            "documents": [
                written.to_manifest_dict() for written in self.written_documents
            ],
        }

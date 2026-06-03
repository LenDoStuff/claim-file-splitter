from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .classifiers import PageClassifier
from .models import ClassificationBatch, DocumentSegment, DocumentType, PageDecision
from .models import PageFeatures
from .models import SplitResult
from .pdf import analyze_pdf, render_pdf_pages, split_pdf


HIGH_CONFIDENCE_BATCH_BOUNDARY = 0.75


@dataclass(frozen=True)
class SplitterConfig:
    output_dir: Path = Path("output")
    batch_size: int = 5
    max_stored_text_chars: int = 12000
    use_pdfplumber_fallback: bool = True
    render_dpi: int = 160
    image_format: str = "jpeg"
    image_quality: int = 85
    keep_page_images: bool = False

    def validated(self) -> "SplitterConfig":
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_stored_text_chars < 1000:
            raise ValueError("max_stored_text_chars must be at least 1000.")
        if self.render_dpi < 72:
            raise ValueError("render_dpi must be at least 72.")
        if self.image_format.lower() not in {"jpeg", "jpg", "png"}:
            raise ValueError("image_format must be 'jpeg' or 'png'.")
        if not 1 <= self.image_quality <= 100:
            raise ValueError("image_quality must be between 1 and 100.")
        return self


class ClaimFileSplitter:
    def __init__(
        self,
        classifier: PageClassifier,
        config: SplitterConfig | None = None,
    ) -> None:
        self.classifier = classifier
        self.config = (config or SplitterConfig()).validated()

    def run(self, input_pdf: str | Path) -> SplitResult:
        source_pdf = Path(input_pdf)
        if not source_pdf.exists():
            raise FileNotFoundError(source_pdf)
        if source_pdf.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF input, got: {source_pdf}")

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        pages = analyze_pdf(
            source_pdf,
            max_stored_text_chars=self.config.max_stored_text_chars,
            use_pdfplumber_fallback=self.config.use_pdfplumber_fallback,
        )
        with ExitStack() as stack:
            render_dir: Path | None = None
            if _classifier_requires_images(self.classifier):
                if self.config.keep_page_images:
                    render_dir = output_dir / "page_images"
                else:
                    render_dir = Path(stack.enter_context(TemporaryDirectory()))

            page_decisions, pages, classification_batches = self._classify_pages(
                source_pdf,
                pages,
                render_dir=render_dir,
            )
            segments = build_segments(page_decisions)
            written_documents = split_pdf(source_pdf, segments, output_dir)
            manifest_path = output_dir / "manifest.json"

            result = SplitResult(
                source_pdf=source_pdf,
                output_dir=output_dir,
                pages=pages,
                page_decisions=page_decisions,
                segments=segments,
                written_documents=written_documents,
                manifest_path=manifest_path,
                classification_batches=classification_batches,
            )
            manifest_path.write_text(
                json.dumps(result.to_manifest_dict(), indent=2),
                encoding="utf-8",
            )
            return result

    def _classify_pages(
        self,
        source_pdf: Path,
        pages: list[PageFeatures],
        *,
        render_dir: Path | None,
    ) -> tuple[list[PageDecision], list[PageFeatures], list[ClassificationBatch]]:
        decisions: list[PageDecision] = []
        batches: list[ClassificationBatch] = []
        requires_images = _classifier_requires_images(self.classifier)

        for start in range(0, len(pages), self.config.batch_size):
            batch = pages[start : start + self.config.batch_size]
            batch_number = (start // self.config.batch_size) + 1
            rolling_context = _build_rolling_context(decisions)

            if requires_images:
                if render_dir is None:
                    raise ValueError("render_dir is required for image classification.")
                rendered_images = render_pdf_pages(
                    source_pdf,
                    [page.page_number for page in batch],
                    render_dir,
                    dpi=self.config.render_dpi,
                    image_format=self.config.image_format,
                    jpeg_quality=self.config.image_quality,
                    keep_paths=self.config.keep_page_images,
                )
                batch = [
                    replace(page, image=rendered_images[page.page_number])
                    for page in batch
                ]
                pages[start : start + len(batch)] = batch

            previous_page = pages[start - 1] if start > 0 else None
            batch_decisions = self.classifier.classify_pages(
                batch,
                previous_page=previous_page,
                rolling_context=rolling_context,
            )
            batch_decisions, reconciliation_messages = _reconcile_batch_boundary(
                batch_decisions,
                decisions,
            )
            decisions.extend(batch_decisions)
            batches.append(
                ClassificationBatch(
                    batch_number=batch_number,
                    start_page=batch[0].page_number,
                    end_page=batch[-1].page_number,
                    page_numbers=[page.page_number for page in batch],
                    rolling_context=rolling_context,
                    reconciliation_messages=reconciliation_messages,
                )
            )

        return _dedupe_and_sort_decisions(decisions, pages), pages, batches


def _classifier_requires_images(classifier: PageClassifier) -> bool:
    return bool(getattr(classifier, "requires_page_images", False))


def _build_rolling_context(decisions: list[PageDecision]) -> dict[str, Any]:
    if not decisions:
        return {
            "open_document": None,
            "recent_page_decisions": [],
            "completed_documents": [],
            "document_type_counts": {},
        }

    segments = build_segments(decisions)
    open_document = segments[-1].to_manifest_dict()
    completed_documents = [
        segment.to_manifest_dict() for segment in segments[:-1][-3:]
    ]
    document_type_counts = Counter(segment.document_type for segment in segments)
    return {
        "open_document": open_document,
        "recent_page_decisions": [
            decision.to_manifest_dict() for decision in decisions[-5:]
        ],
        "completed_documents": completed_documents,
        "document_type_counts": dict(document_type_counts),
    }


def _reconcile_batch_boundary(
    batch_decisions: list[PageDecision],
    accumulated_decisions: list[PageDecision],
) -> tuple[list[PageDecision], list[str]]:
    normalized = [decision.normalized() for decision in batch_decisions]
    if not normalized or not accumulated_decisions:
        return normalized, []

    first = normalized[0]
    previous = accumulated_decisions[-1].normalized()
    messages: list[str] = []

    if not first.starts_new_document:
        if first.document_type != previous.document_type:
            reason = (
                "Batch boundary reconciliation: first page continued the prior "
                f"document, so type was inherited from page {previous.page_number}."
            )
            first = replace(
                first,
                document_type=previous.document_type,
                reason=_append_reason(first.reason, reason),
            )
            messages.append(reason)
    elif first.confidence < HIGH_CONFIDENCE_BATCH_BOUNDARY:
        reason = (
            "Batch boundary reconciliation: low-confidence new boundary on first "
            f"batch page was treated as continuation of page {previous.page_number}."
        )
        first = replace(
            first,
            starts_new_document=False,
            document_type=previous.document_type,
            reason=_append_reason(first.reason, reason),
        )
        messages.append(reason)
    else:
        messages.append(
            "Batch boundary reconciliation: preserved high-confidence new boundary "
            f"on page {first.page_number}."
        )

    normalized[0] = first
    return normalized, messages


def _append_reason(existing: str, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing} {addition}"


def build_segments(page_decisions: list[PageDecision]) -> list[DocumentSegment]:
    if not page_decisions:
        return []

    sorted_decisions = sorted(page_decisions, key=lambda decision: decision.page_number)
    segments: list[DocumentSegment] = []
    current: list[PageDecision] = []

    for decision in sorted_decisions:
        decision = decision.normalized()
        starts = decision.starts_new_document
        if current and _should_force_boundary(current[-1], decision):
            starts = True

        if current and starts:
            segments.append(_segment_from_decisions(len(segments) + 1, current))
            current = [decision]
        else:
            current.append(decision)

    if current:
        segments.append(_segment_from_decisions(len(segments) + 1, current))

    return segments


def _dedupe_and_sort_decisions(
    decisions: list[PageDecision],
    pages: list[PageFeatures],
) -> list[PageDecision]:
    by_page = {decision.page_number: decision.normalized() for decision in decisions}
    repaired: list[PageDecision] = []
    for page in pages:
        decision = by_page.get(page.page_number)
        if decision is None:
            decision = PageDecision(
                page_number=page.page_number,
                document_type="photos" if page.is_image_only else "other",
                starts_new_document=page.page_number == 1,
                title="",
                confidence=0.2,
                reason="Missing classifier decision.",
            )
        if page.page_number == 1 and not decision.starts_new_document:
            decision = PageDecision(
                page_number=decision.page_number,
                document_type=decision.document_type,
                starts_new_document=True,
                title=decision.title,
                confidence=decision.confidence,
                reason=decision.reason,
            )
        repaired.append(decision)
    return repaired


def _should_force_boundary(previous: PageDecision, current: PageDecision) -> bool:
    if current.document_type == previous.document_type:
        return False
    if current.document_type == "other" or previous.document_type == "other":
        return current.confidence >= 0.75
    return current.confidence >= 0.5


def _segment_from_decisions(
    segment_id: int,
    decisions: list[PageDecision],
) -> DocumentSegment:
    document_type = _majority_document_type(decisions)
    reasons = _unique_reasons(decisions)
    title = _first_title(decisions) or document_type.replace("_", " ").title()
    confidence = sum(decision.confidence for decision in decisions) / len(decisions)
    return DocumentSegment(
        segment_id=segment_id,
        document_type=document_type,
        start_page=decisions[0].page_number,
        end_page=decisions[-1].page_number,
        title=title,
        confidence=confidence,
        reasons=reasons,
    )


def _majority_document_type(decisions: list[PageDecision]) -> DocumentType:
    counts = Counter(decision.document_type for decision in decisions)
    return counts.most_common(1)[0][0]


def _unique_reasons(decisions: list[PageDecision], limit: int = 5) -> list[str]:
    reasons: list[str] = []
    for decision in decisions:
        reason = decision.reason.strip()
        if reason and reason not in reasons:
            reasons.append(reason)
        if len(reasons) == limit:
            break
    return reasons


def _first_title(decisions: list[PageDecision]) -> str:
    for decision in decisions:
        if decision.title.strip():
            return decision.title.strip()
    return ""

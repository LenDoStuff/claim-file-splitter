from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .classifiers import PageClassifier
from .models import DocumentSegment, DocumentType, PageDecision, PageFeatures
from .models import SplitResult
from .pdf import analyze_pdf, split_pdf


@dataclass(frozen=True)
class SplitterConfig:
    output_dir: Path = Path("output")
    batch_size: int = 20
    max_stored_text_chars: int = 12000
    use_pdfplumber_fallback: bool = True

    def validated(self) -> "SplitterConfig":
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_stored_text_chars < 1000:
            raise ValueError("max_stored_text_chars must be at least 1000.")
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
        page_decisions = self._classify_pages(pages)
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
        )
        manifest_path.write_text(
            json.dumps(result.to_manifest_dict(), indent=2),
            encoding="utf-8",
        )
        return result

    def _classify_pages(self, pages: list[PageFeatures]) -> list[PageDecision]:
        decisions: list[PageDecision] = []
        for start in range(0, len(pages), self.config.batch_size):
            batch = pages[start : start + self.config.batch_size]
            previous_page = pages[start - 1] if start > 0 else None
            decisions.extend(
                self.classifier.classify_pages(batch, previous_page=previous_page)
            )
        return _dedupe_and_sort_decisions(decisions, pages)


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

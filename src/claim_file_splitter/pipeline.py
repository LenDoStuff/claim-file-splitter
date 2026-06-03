from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from .classifiers import rule_based_classify_pages
from .customization import DEFAULT_BATCH_SIZE
from .models import make_decision, result_manifest, segment_manifest
from .pdf import analyze_pdf, render_pdf_pages, split_pdf


HIGH_CONFIDENCE_BATCH_BOUNDARY = 0.75


def split_claim_file(
    input_pdf: str | Path,
    *,
    output_dir: str | Path = "output",
    classify_pages: Callable[..., list[dict[str, Any]]] = rule_based_classify_pages,
    requires_page_images: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_stored_text_chars: int = 12000,
    use_pdfplumber_fallback: bool = True,
    render_dpi: int = 160,
    image_format: str = "jpeg",
    image_quality: int = 85,
    keep_page_images: bool = False,
) -> dict[str, Any]:
    source_pdf = Path(input_pdf)
    if not source_pdf.exists():
        raise FileNotFoundError(source_pdf)
    if source_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF input, got: {source_pdf}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pages = analyze_pdf(
        source_pdf,
        max_stored_text_chars=max_stored_text_chars,
        use_pdfplumber_fallback=use_pdfplumber_fallback,
    )

    with ExitStack() as stack:
        render_dir = None
        if requires_page_images:
            render_dir = output_path / "page_images" if keep_page_images else Path(
                stack.enter_context(TemporaryDirectory())
            )

        page_decisions, pages, classification_batches = classify_page_batches(
            source_pdf,
            pages,
            classify_pages=classify_pages,
            requires_page_images=requires_page_images,
            render_dir=render_dir,
            batch_size=batch_size,
            render_dpi=render_dpi,
            image_format=image_format,
            image_quality=image_quality,
            keep_page_images=keep_page_images,
        )
        segments = build_segments(page_decisions)
        written_documents = split_pdf(source_pdf, segments, output_path)

        result = {
            "source_pdf": source_pdf,
            "output_dir": output_path,
            "pages": pages,
            "page_decisions": page_decisions,
            "classification_batches": classification_batches,
            "segments": segments,
            "written_documents": written_documents,
            "manifest_path": output_path / "manifest.json",
        }
        result["manifest_path"].write_text(
            json.dumps(result_manifest(result), indent=2),
            encoding="utf-8",
        )
        return result


def classify_page_batches(
    source_pdf: Path,
    pages: list[dict[str, Any]],
    *,
    classify_pages: Callable[..., list[dict[str, Any]]],
    requires_page_images: bool,
    render_dir: Path | None,
    batch_size: int,
    render_dpi: int,
    image_format: str,
    image_quality: int,
    keep_page_images: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = []
    batches = []

    for start in range(0, len(pages), batch_size):
        batch = pages[start : start + batch_size]
        rolling_context = build_rolling_context(decisions)

        if requires_page_images:
            rendered_images = render_pdf_pages(
                source_pdf,
                [page["page_number"] for page in batch],
                render_dir,
                dpi=render_dpi,
                image_format=image_format,
                jpeg_quality=image_quality,
                keep_paths=keep_page_images,
            )
            batch = [
                {**page, "image": rendered_images[page["page_number"]]}
                for page in batch
            ]
            pages[start : start + len(batch)] = batch

        batch_decisions = classify_pages(
            batch,
            previous_page=pages[start - 1] if start > 0 else None,
            rolling_context=rolling_context,
        )
        batch_decisions, reconciliation_messages = reconcile_batch_boundary(
            batch_decisions,
            decisions,
        )
        decisions.extend(batch_decisions)
        batches.append(
            {
                "batch_number": (start // batch_size) + 1,
                "start_page": batch[0]["page_number"],
                "end_page": batch[-1]["page_number"],
                "page_numbers": [page["page_number"] for page in batch],
                "rolling_context": rolling_context,
                "reconciliation_messages": reconciliation_messages,
            }
        )

    return dedupe_and_sort_decisions(decisions, pages), pages, batches


def build_rolling_context(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    if not decisions:
        return {
            "open_document": None,
            "recent_page_decisions": [],
            "completed_documents": [],
            "document_type_counts": {},
        }

    segments = build_segments(decisions)
    document_type_counts = Counter(segment["document_type"] for segment in segments)
    return {
        "open_document": segment_manifest(segments[-1]),
        "recent_page_decisions": decisions[-5:],
        "completed_documents": [
            segment_manifest(segment) for segment in segments[:-1][-3:]
        ],
        "document_type_counts": dict(document_type_counts),
    }


def reconcile_batch_boundary(
    batch_decisions: list[dict[str, Any]],
    accumulated_decisions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    if not batch_decisions or not accumulated_decisions:
        return batch_decisions, []

    first = batch_decisions[0]
    previous = accumulated_decisions[-1]
    messages = []

    if not first["starts_new_document"]:
        if first["document_type"] != previous["document_type"]:
            reason = (
                "Batch boundary reconciliation: first page continued the prior "
                f"document, so type was inherited from page {previous['page_number']}."
            )
            first = {
                **first,
                "document_type": previous["document_type"],
                "reason": append_reason(first["reason"], reason),
            }
            messages.append(reason)
    elif first["confidence"] < HIGH_CONFIDENCE_BATCH_BOUNDARY:
        reason = (
            "Batch boundary reconciliation: low-confidence new boundary on first "
            f"batch page was treated as continuation of page {previous['page_number']}."
        )
        first = {
            **first,
            "starts_new_document": False,
            "document_type": previous["document_type"],
            "reason": append_reason(first["reason"], reason),
        }
        messages.append(reason)
    else:
        messages.append(
            "Batch boundary reconciliation: preserved high-confidence new boundary "
            f"on page {first['page_number']}."
        )

    return [first, *batch_decisions[1:]], messages


def build_segments(page_decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments = []
    current = []

    for decision in sorted(page_decisions, key=lambda item: item["page_number"]):
        starts = decision["starts_new_document"]
        if current and should_force_boundary(current[-1], decision):
            starts = True

        if current and starts:
            segments.append(segment_from_decisions(len(segments) + 1, current))
            current = [decision]
        else:
            current.append(decision)

    if current:
        segments.append(segment_from_decisions(len(segments) + 1, current))

    return segments


def dedupe_and_sort_decisions(
    decisions: list[dict[str, Any]],
    pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_page = {decision["page_number"]: decision for decision in decisions}
    repaired = []
    for page in pages:
        decision = by_page.get(page["page_number"])
        if decision is None:
            decision = make_decision(
                page["page_number"],
                "photos" if page["is_image_only"] else "other",
                page["page_number"] == 1,
                confidence=0.2,
                reason="Missing classifier decision.",
            )
        if page["page_number"] == 1 and not decision["starts_new_document"]:
            decision = {**decision, "starts_new_document": True}
        repaired.append(decision)
    return repaired


def should_force_boundary(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> bool:
    if current["document_type"] == previous["document_type"]:
        return False
    if current["document_type"] == "other" or previous["document_type"] == "other":
        return current["confidence"] >= 0.75
    return current["confidence"] >= 0.5


def segment_from_decisions(
    segment_id: int,
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(decision["document_type"] for decision in decisions)
    document_type = counts.most_common(1)[0][0]
    reasons = unique_reasons(decisions)
    return {
        "segment_id": segment_id,
        "document_type": document_type,
        "start_page": decisions[0]["page_number"],
        "end_page": decisions[-1]["page_number"],
        "title": first_title(decisions) or document_type.replace("_", " ").title(),
        "confidence": sum(decision["confidence"] for decision in decisions)
        / len(decisions),
        "reasons": reasons,
    }


def unique_reasons(decisions: list[dict[str, Any]], limit: int = 5) -> list[str]:
    reasons = []
    for decision in decisions:
        reason = decision["reason"].strip()
        if reason and reason not in reasons:
            reasons.append(reason)
        if len(reasons) == limit:
            break
    return reasons


def first_title(decisions: list[dict[str, Any]]) -> str:
    for decision in decisions:
        if decision["title"].strip():
            return decision["title"].strip()
    return ""


def append_reason(existing: str, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing} {addition}"

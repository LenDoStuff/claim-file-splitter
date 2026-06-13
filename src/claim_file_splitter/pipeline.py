from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from .classifiers import azure_classify_pages, make_azure_openai_client
from .classifiers import rule_based_classify_pages
from .customization import ClaimSplitterConfig, category_prefixes, default_config
from .customization import resolve_config
from .models import ClaimSplitResult, make_decision, result_manifest
from .models import segment_manifest, typed_result
from .pdf import analyze_pdf, render_pdf_pages, split_pdf


def split_claim_file_azure(
    input_pdf: str | Path,
    *,
    output_dir: str | Path = "output",
    config: ClaimSplitterConfig | dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    project_endpoint: str | None = None,
    deployment: str | None = None,
    batch_size: int | None = None,
    render_dpi: int | None = None,
    image_format: str | None = None,
    image_quality: int | None = None,
    image_detail: str | None = None,
    keep_page_images: bool | None = None,
    max_stored_text_chars: int | None = None,
    use_pdfplumber_fallback: bool | None = None,
    client: Any | None = None,
) -> ClaimSplitResult:
    active_config = resolve_config(
        config=config,
        config_path=config_path,
        project_endpoint=project_endpoint,
        deployment=deployment,
        batch_size=batch_size,
        render_dpi=render_dpi,
        image_format=image_format,
        image_quality=image_quality,
        image_detail=image_detail,
        keep_page_images=keep_page_images,
        max_stored_text_chars=max_stored_text_chars,
        use_pdfplumber_fallback=use_pdfplumber_fallback,
    )
    if not active_config.azure.deployment:
        raise ValueError(
            "Azure deployment is required. Pass deployment, set it in config, "
            "or set AZURE_OPENAI_DEPLOYMENT."
        )

    project_client = None
    openai_client = client
    try:
        if openai_client is None:
            if not active_config.azure.project_endpoint:
                raise ValueError(
                    "Azure project endpoint is required. Pass project_endpoint, "
                    "set it in config, or set AZURE_AI_PROJECT_ENDPOINT."
                )
            project_client, openai_client = make_azure_openai_client(
                active_config.azure.project_endpoint
            )

        def classify_pages(batch, *, previous_page=None, rolling_context=None):
            return azure_classify_pages(
                openai_client,
                active_config.azure.deployment,
                batch,
                config=active_config,
                previous_page=previous_page,
                rolling_context=rolling_context,
            )

        return run_split_pipeline(
            input_pdf,
            output_dir=output_dir,
            config=active_config,
            classify_pages=classify_pages,
            requires_page_images=True,
        )
    finally:
        if client is None:
            close_clients(openai_client, project_client)


def split_claim_file_rules(
    input_pdf: str | Path,
    *,
    output_dir: str | Path = "output",
    config: ClaimSplitterConfig | dict[str, Any] | None = None,
    config_path: str | Path | None = None,
    batch_size: int | None = None,
    render_dpi: int | None = None,
    image_format: str | None = None,
    image_quality: int | None = None,
    keep_page_images: bool | None = None,
    max_stored_text_chars: int | None = None,
    use_pdfplumber_fallback: bool | None = None,
) -> ClaimSplitResult:
    active_config = resolve_config(
        config=config,
        config_path=config_path,
        batch_size=batch_size,
        render_dpi=render_dpi,
        image_format=image_format,
        image_quality=image_quality,
        keep_page_images=keep_page_images,
        max_stored_text_chars=max_stored_text_chars,
        use_pdfplumber_fallback=use_pdfplumber_fallback,
    )

    def classify_pages(batch, *, previous_page=None, rolling_context=None):
        return rule_based_classify_pages(
            batch,
            config=active_config,
            previous_page=previous_page,
            rolling_context=rolling_context,
        )

    return run_split_pipeline(
        input_pdf,
        output_dir=output_dir,
        config=active_config,
        classify_pages=classify_pages,
        requires_page_images=False,
    )


def run_split_pipeline(
    input_pdf: str | Path,
    *,
    output_dir: str | Path,
    config: ClaimSplitterConfig,
    classify_pages: Callable[..., list[dict[str, Any]]],
    requires_page_images: bool,
) -> ClaimSplitResult:
    source_pdf = Path(input_pdf)
    if not source_pdf.exists():
        raise FileNotFoundError(source_pdf)
    if source_pdf.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF input, got: {source_pdf}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pages = analyze_pdf(
        source_pdf,
        max_stored_text_chars=config.splitter.max_stored_text_chars,
        use_pdfplumber_fallback=config.splitter.use_pdfplumber_fallback,
    )

    with ExitStack() as stack:
        render_dir = None
        if requires_page_images:
            render_dir = (
                output_path / "page_images"
                if config.rendering.keep_page_images
                else Path(stack.enter_context(TemporaryDirectory()))
            )

        page_decisions, pages, classification_batches = classify_page_batches(
            source_pdf,
            pages,
            classify_pages=classify_pages,
            requires_page_images=requires_page_images,
            render_dir=render_dir,
            config=config,
        )
        segments = build_segments(page_decisions, config)
        written_documents = split_pdf(
            source_pdf,
            segments,
            output_path,
            category_prefixes(config),
        )

        raw_result = {
            "source_pdf": source_pdf,
            "output_dir": output_path,
            "pages": pages,
            "page_decisions": page_decisions,
            "classification_batches": classification_batches,
            "segments": segments,
            "written_documents": written_documents,
            "manifest_path": output_path / "manifest.json",
        }
        result = typed_result(raw_result)
        result.manifest_path.write_text(
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
    config: ClaimSplitterConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = []
    batches = []

    for start in range(0, len(pages), config.splitter.batch_size):
        batch = pages[start : start + config.splitter.batch_size]
        rolling_context = build_rolling_context(decisions, config)

        if requires_page_images:
            rendered_images = render_pdf_pages(
                source_pdf,
                [page["page_number"] for page in batch],
                render_dir,
                dpi=config.rendering.dpi,
                image_format=config.rendering.image_format,
                jpeg_quality=config.rendering.image_quality,
                keep_paths=config.rendering.keep_page_images,
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
            config,
        )
        decisions.extend(batch_decisions)
        batches.append(
            {
                "batch_number": (start // config.splitter.batch_size) + 1,
                "start_page": batch[0]["page_number"],
                "end_page": batch[-1]["page_number"],
                "page_numbers": [page["page_number"] for page in batch],
                "rolling_context": rolling_context,
                "reconciliation_messages": reconciliation_messages,
            }
        )

    return dedupe_and_sort_decisions(decisions, pages, config), pages, batches


def build_rolling_context(
    decisions: list[dict[str, Any]],
    config: ClaimSplitterConfig | None = None,
) -> dict[str, Any]:
    active_config = config or default_config()
    if not decisions:
        return {
            "open_document": None,
            "recent_page_decisions": [],
            "completed_documents": [],
            "document_type_counts": {},
        }

    segments = build_segments(decisions, active_config)
    document_type_counts = Counter(segment["document_type"] for segment in segments)
    recent_limit = active_config.splitter.recent_page_decision_limit
    completed_limit = active_config.splitter.completed_document_limit
    recent_decisions = [] if recent_limit == 0 else decisions[-recent_limit:]
    completed_segments = [] if completed_limit == 0 else segments[:-1][-completed_limit:]
    return {
        "open_document": segment_manifest(segments[-1]),
        "recent_page_decisions": recent_decisions,
        "completed_documents": [
            segment_manifest(segment)
            for segment in completed_segments
        ],
        "document_type_counts": dict(document_type_counts),
    }


def reconcile_batch_boundary(
    batch_decisions: list[dict[str, Any]],
    accumulated_decisions: list[dict[str, Any]],
    config: ClaimSplitterConfig | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    active_config = config or default_config()
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
    elif (
        first["confidence"]
        < active_config.splitter.high_confidence_batch_boundary
    ):
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


def build_segments(
    page_decisions: list[dict[str, Any]],
    config: ClaimSplitterConfig | None = None,
) -> list[dict[str, Any]]:
    active_config = config or default_config()
    segments = []
    current = []

    for decision in sorted(page_decisions, key=lambda item: item["page_number"]):
        starts = decision["starts_new_document"]
        if current and should_force_boundary(current[-1], decision, active_config):
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
    config: ClaimSplitterConfig,
) -> list[dict[str, Any]]:
    by_page = {decision["page_number"]: decision for decision in decisions}
    repaired = []
    for page in pages:
        decision = by_page.get(page["page_number"])
        if decision is None:
            document_type = (
                "photos"
                if page["is_image_only"]
                and any(category.name == "photos" for category in config.categories)
                else config.default_document_type
            )
            decision = make_decision(
                page["page_number"],
                document_type,
                page["page_number"] == 1,
                config=config,
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
    config: ClaimSplitterConfig | None = None,
) -> bool:
    active_config = config or default_config()
    if current["document_type"] == previous["document_type"]:
        return False
    if (
        current["document_type"] == active_config.default_document_type
        or previous["document_type"] == active_config.default_document_type
    ):
        return current["confidence"] >= (
            active_config.splitter.other_type_boundary_confidence
        )
    return current["confidence"] >= (
        active_config.splitter.type_change_boundary_confidence
    )


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


def close_clients(*clients) -> None:
    for client in clients:
        close = getattr(client, "close", None)
        if callable(close):
            close()

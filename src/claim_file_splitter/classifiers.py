from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from .customization import BatchClassificationOutput, IMAGE_DETAIL, SYSTEM_PROMPT
from .customization import RULE_KEYWORDS, USER_PROMPT
from .models import DOCUMENT_TYPES, make_decision
from .models import normalize_document_type, page_image_prompt


def rule_based_classify_pages(
    pages: Sequence[dict[str, Any]],
    *,
    previous_page: dict[str, Any] | None = None,
    rolling_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    decisions = []
    previous_type = classify_page_by_rules(previous_page)[0] if previous_page else None

    for page in pages:
        document_type, confidence, reason = classify_page_by_rules(page)
        starts_new_document = page["page_number"] == 1 or previous_type is None
        if previous_type is not None and document_type != previous_type:
            starts_new_document = True

        decisions.append(
            make_decision(
                page["page_number"],
                document_type,
                starts_new_document,
                title=first_line(page["text"]),
                confidence=confidence,
                reason=reason,
            )
        )
        previous_type = document_type

    return decisions


def classify_page_by_rules(page: dict[str, Any]) -> tuple[str, float, str]:
    if page["is_image_only"]:
        return "photos", 0.55, "Image-only page with no extractable text."

    text = page["text"].lower()
    scores = {
        document_type: sum(1 for keyword in keywords if keyword in text)
        for document_type, keywords in RULE_KEYWORDS.items()
    }
    scores = {document_type: score for document_type, score in scores.items() if score}
    if not scores:
        return "other", 0.25, "No strong claim-document keywords matched."

    document_type = max(scores, key=scores.get)
    score = scores[document_type]
    return document_type, min(0.95, 0.45 + (score * 0.12)), (
        f"Matched {score} keyword signal(s)."
    )


def make_azure_openai_client(project_endpoint: str) -> tuple[Any, Any]:
    from azure.ai.projects import AIProjectClient
    from azure.identity import DefaultAzureCredential

    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=DefaultAzureCredential(),
    )
    return project_client, project_client.get_openai_client()


def azure_classify_pages(
    client: Any,
    deployment: str,
    pages: Sequence[dict[str, Any]],
    *,
    previous_page: dict[str, Any] | None = None,
    rolling_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    missing_images = [page["page_number"] for page in pages if page.get("image") is None]
    if missing_images:
        raise ValueError(
            "Azure image classification requires rendered page images. "
            f"Missing images for page(s): {missing_images}"
        )

    prompt = build_azure_prompt(pages, rolling_context)
    parsed = call_azure_model(
        client,
        deployment,
        prompt,
        pages,
    )
    decisions = decisions_from_structured_output(parsed)
    return repair_decision_list(decisions, pages)


def build_azure_prompt(
    pages: Sequence[dict[str, Any]],
    rolling_context: dict[str, Any] | None,
) -> str:
    return json.dumps(
        {
            "allowed_document_types": list(DOCUMENT_TYPES),
            "instructions": USER_PROMPT,
            "rolling_context": rolling_context or {},
            "target_pages": [page_image_prompt(page) for page in pages],
        },
        ensure_ascii=True,
    )


def call_azure_model(
    client: Any,
    deployment: str,
    prompt: str,
    pages: Sequence[dict[str, Any]],
) -> BatchClassificationOutput:
    if not hasattr(client, "responses") or not hasattr(client.responses, "parse"):
        raise ValueError("OpenAI client must provide responses.parse.")

    response = client.responses.parse(
        model=deployment,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    *[
                        {
                            "type": "input_image",
                            "image_url": page["image"]["data_uri"],
                            "detail": IMAGE_DETAIL,
                        }
                        for page in pages
                    ],
                ],
            },
        ],
        text_format=BatchClassificationOutput,
        temperature=0,
    )
    return response.output_parsed


def decisions_from_structured_output(
    output: BatchClassificationOutput,
) -> list[dict[str, Any]]:
    decisions = []
    for item in output.pages:
        decisions.append(
            make_decision(
                item.page,
                normalize_document_type(item.document_type.value),
                item.starts_new_document,
                title=item.title,
                confidence=item.confidence,
                reason=item.reason,
            )
        )
    return decisions


def repair_decision_list(
    decisions: Sequence[dict[str, Any]],
    pages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_page = {decision["page_number"]: decision for decision in decisions}
    repaired = []
    for page in pages:
        decision = by_page.get(page["page_number"])
        if decision is None:
            decision = make_decision(
                page["page_number"],
                "photos" if page["is_image_only"] else "other",
                False,
                title=first_line(page["text"]),
                confidence=0.2,
                reason="Classifier omitted this page; fallback decision inserted.",
            )
        if page["page_number"] == 1:
            decision = {**decision, "starts_new_document": True}
        repaired.append(decision)
    return repaired


def first_line(text: str, limit: int = 90) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""

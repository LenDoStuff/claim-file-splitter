from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from .customization import ClaimSplitterConfig, make_batch_classification_output_model
from .models import make_decision, normalize_document_type


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
    config: ClaimSplitterConfig,
    rolling_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    missing_images = [page["page_number"] for page in pages if page.get("image") is None]
    if missing_images:
        raise ValueError(
            "Azure image classification requires rendered page images. "
            f"Missing images for page(s): {missing_images}"
        )

    prompt = build_azure_prompt(pages, rolling_context, config)
    text_format = make_batch_classification_output_model(config)
    parsed = call_azure_model(
        client,
        deployment,
        prompt,
        pages,
        config=config,
        text_format=text_format,
    )
    decisions = decisions_from_structured_output(parsed, config)
    return repair_decision_list(decisions, pages, config)


def build_azure_prompt(
    pages: Sequence[dict[str, Any]],
    rolling_context: dict[str, Any] | None,
    config: ClaimSplitterConfig,
) -> str:
    return json.dumps(
        {
            "allowed_document_types": [
                {
                    "name": category.name,
                    "description": category.description,
                }
                for category in config.categories
            ],
            "instructions": config.prompts.user_prompt,
            "rolling_context": rolling_context or {},
            "target_pages": [
                {
                    "page": page["page_number"],
                    "rendered_image": page_prompt_image_metadata(page.get("image")),
                }
                for page in pages
            ],
        },
        ensure_ascii=True,
    )


def page_prompt_image_metadata(image: dict[str, Any] | None) -> dict[str, Any] | None:
    if image is None:
        return None
    return {
        "page": image["page_number"],
        "mime_type": image["mime_type"],
        "width_px": image["width_px"],
        "height_px": image["height_px"],
        "byte_size": image["byte_size"],
    }


def call_azure_model(
    client: Any,
    deployment: str,
    prompt: str,
    pages: Sequence[dict[str, Any]],
    *,
    config: ClaimSplitterConfig,
    text_format: type[BaseModel],
) -> BaseModel:
    if not hasattr(client, "responses") or not hasattr(client.responses, "parse"):
        raise ValueError("OpenAI client must provide responses.parse.")

    response = client.responses.parse(
        model=deployment,
        input=[
            {
                "role": "system",
                "content": config.prompts.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    *[
                        {
                            "type": "input_image",
                            "image_url": page["image"]["data_uri"],
                            "detail": config.rendering.image_detail,
                        }
                        for page in pages
                    ],
                ],
            },
        ],
        text_format=text_format,
        temperature=config.azure.temperature,
    )
    return response.output_parsed


def decisions_from_structured_output(
    output: BaseModel,
    config: ClaimSplitterConfig,
) -> list[dict[str, Any]]:
    decisions = []
    for item in output.pages:  # type: ignore[attr-defined]
        document_type = item.document_type
        if hasattr(document_type, "value"):
            document_type = document_type.value
        decisions.append(
            make_decision(
                item.page,
                normalize_document_type(str(document_type), config),
                item.starts_new_document,
                config=config,
                title=item.title,
                confidence=item.confidence,
                reason=item.reason,
            )
        )
    return decisions


def repair_decision_list(
    decisions: Sequence[dict[str, Any]],
    pages: Sequence[dict[str, Any]],
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
                False,
                config=config,
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

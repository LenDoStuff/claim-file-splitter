from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol

from .models import DOCUMENT_TYPES, DocumentType, PageDecision, PageFeatures
from .models import normalize_document_type


class PageClassifier(Protocol):
    requires_page_images: bool

    def classify_pages(
        self,
        pages: Sequence[PageFeatures],
        *,
        previous_page: PageFeatures | None = None,
        rolling_context: dict[str, Any] | None = None,
    ) -> list[PageDecision]:
        """Return one classification and boundary decision for each page."""


class RuleBasedPageClassifier:
    """Deterministic fallback used for tests and credential-free dry runs."""

    requires_page_images = False

    _keywords: dict[DocumentType, tuple[str, ...]] = {
        "repair_invoices": (
            "repair invoice",
            "auto repair",
            "body shop",
            "parts",
            "labor",
            "amount due",
            "invoice number",
        ),
        "appraisals": (
            "appraisal",
            "valuation",
            "estimate of damages",
            "damage estimate",
            "vehicle value",
        ),
        "communications": (
            "from:",
            "to:",
            "sent:",
            "subject:",
            "email thread",
            "dear ",
            "regards",
        ),
        "police_reports": (
            "police report",
            "incident report",
            "crash report",
            "officer",
            "case number",
            "department",
        ),
        "payments": (
            "payment",
            "check number",
            "paid",
            "remittance",
            "settlement payment",
            "payment notice",
        ),
        "medical": (
            "medical",
            "patient",
            "diagnosis",
            "treatment",
            "physician",
            "hospital",
            "clinic",
        ),
        "legal_correspondence": (
            "law office",
            "attorney",
            "legal",
            "demand letter",
            "counsel",
            "litigation",
            "subpoena",
        ),
        "photos": ("photograph", "photo log", "image section", "scene photos"),
        "other": (),
    }

    def classify_pages(
        self,
        pages: Sequence[PageFeatures],
        *,
        previous_page: PageFeatures | None = None,
        rolling_context: dict[str, Any] | None = None,
    ) -> list[PageDecision]:
        decisions: list[PageDecision] = []
        previous_type = (
            self._classify_page(previous_page)[0] if previous_page is not None else None
        )

        for page in pages:
            document_type, confidence, reason = self._classify_page(page)
            starts_new_document = page.page_number == 1 or previous_type is None
            if previous_type is not None and document_type != previous_type:
                starts_new_document = True

            decisions.append(
                PageDecision(
                    page_number=page.page_number,
                    document_type=document_type,
                    starts_new_document=starts_new_document,
                    title=_first_line(page.text),
                    confidence=confidence,
                    reason=reason,
                )
            )
            previous_type = document_type

        return decisions

    def _classify_page(self, page: PageFeatures) -> tuple[DocumentType, float, str]:
        if page.is_image_only:
            return "photos", 0.55, "Image-only page with no extractable text."

        text = page.text.lower()
        scores: dict[DocumentType, int] = {}
        for document_type, keywords in self._keywords.items():
            if document_type == "other":
                continue
            score = sum(1 for keyword in keywords if keyword in text)
            if score:
                scores[document_type] = score

        if not scores:
            return "other", 0.25, "No strong claim-document keywords matched."

        document_type = max(scores, key=scores.get)
        score = scores[document_type]
        confidence = min(0.95, 0.45 + (score * 0.12))
        return document_type, confidence, f"Matched {score} keyword signal(s)."


class AzureProjectPageClassifier:
    """Classify page batches through an Azure AI Projects OpenAI client."""

    requires_page_images = True

    def __init__(
        self,
        *,
        project_endpoint: str,
        deployment: str,
        client: Any | None = None,
        use_responses_api: bool = True,
    ) -> None:
        self.deployment = deployment
        self.use_responses_api = use_responses_api
        self._project_client: Any | None = None

        if client is not None:
            self._client = client
            return

        from azure.ai.projects import AIProjectClient
        from azure.identity import DefaultAzureCredential

        self._project_client = AIProjectClient(
            endpoint=project_endpoint,
            credential=DefaultAzureCredential(),
        )
        self._client = self._project_client.get_openai_client()

    def close(self) -> None:
        for client in (getattr(self, "_client", None), self._project_client):
            close = getattr(client, "close", None)
            if callable(close):
                close()

    def classify_pages(
        self,
        pages: Sequence[PageFeatures],
        *,
        previous_page: PageFeatures | None = None,
        rolling_context: dict[str, Any] | None = None,
    ) -> list[PageDecision]:
        if not pages:
            return []

        missing_images = [page.page_number for page in pages if page.image is None]
        if missing_images:
            raise ValueError(
                "Azure image classification requires rendered page images. "
                f"Missing images for page(s): {missing_images}"
            )

        prompt = self._build_prompt(pages, rolling_context)
        raw_response = self._call_model(prompt, pages)
        payload = _load_json_object(raw_response)
        decisions = _decisions_from_payload(payload)
        return _repair_decision_list(decisions, pages)

    def _build_prompt(
        self,
        pages: Sequence[PageFeatures],
        rolling_context: dict[str, Any] | None,
    ) -> str:
        payload = {
            "allowed_document_types": list(DOCUMENT_TYPES),
            "instructions": [
                "Analyze the rendered page images as part of one insurance claim file.",
                "The attached images are the authoritative page input.",
                "Use only page numbers, image metadata, and rolling context for non-image context.",
                "Set starts_new_document to true when the page begins a new logical document.",
                "Set starts_new_document to false when a batch's first page continues the open document from rolling_context.",
                "Keep multi-page documents together even when headers repeat.",
                "Classify damage-photo or image-section pages as photos when appropriate.",
                "Classify only the pages listed in target_pages, not pages mentioned in rolling_context.",
                "Return only valid JSON with a top-level pages array.",
            ],
            "rolling_context": rolling_context or {},
            "target_pages": [page.to_image_prompt_dict() for page in pages],
            "response_shape": {
                "pages": [
                    {
                        "page": "integer page number",
                        "document_type": "one allowed document type",
                        "starts_new_document": "boolean",
                        "title": "short title",
                        "confidence": "number from 0 to 1",
                        "reason": "short reason",
                    }
                ]
            },
        }
        return json.dumps(payload, ensure_ascii=True)

    def _call_model(self, prompt: str, pages: Sequence[PageFeatures]) -> str:
        system = (
            "You are a claim-file document boundary detector and classifier. "
            "Return compact, valid JSON only."
        )
        image_parts = [
            {"type": "input_image", "image_url": page.image.data_uri}
            for page in pages
            if page.image is not None
        ]

        if self.use_responses_api and hasattr(self._client, "responses"):
            response = self._client.responses.create(
                model=self.deployment,
                input=[
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": system}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            *image_parts,
                        ],
                    },
                ],
                temperature=0,
            )
            text = _extract_response_text(response)
            if text:
                return text

        chat_image_parts = [
            {"type": "image_url", "image_url": {"url": page.image.data_uri}}
            for page in pages
            if page.image is not None
        ]
        completion = self._client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *chat_image_parts,
                    ],
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return completion.choices[0].message.content or "{}"


def _extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        data = model_dump()
    elif isinstance(response, dict):
        data = response
    else:
        return str(response)

    chunks: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            text = content.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


def _load_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Classifier returned JSON that is not an object.")
    return payload


def _decisions_from_payload(payload: dict[str, Any]) -> list[PageDecision]:
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise ValueError("Classifier response must contain a pages array.")

    decisions: list[PageDecision] = []
    for item in pages:
        if not isinstance(item, dict):
            continue
        page_number = item.get("page", item.get("page_number"))
        try:
            page_number = int(page_number)
        except (TypeError, ValueError):
            continue
        decisions.append(
            PageDecision(
                page_number=page_number,
                document_type=normalize_document_type(item.get("document_type")),
                starts_new_document=bool(item.get("starts_new_document")),
                title=str(item.get("title") or ""),
                confidence=float(item.get("confidence") or 0.0),
                reason=str(item.get("reason") or ""),
            ).normalized()
        )
    return decisions


def _repair_decision_list(
    decisions: Sequence[PageDecision],
    pages: Sequence[PageFeatures],
) -> list[PageDecision]:
    by_page = {decision.page_number: decision.normalized() for decision in decisions}
    repaired: list[PageDecision] = []
    for page in pages:
        decision = by_page.get(page.page_number)
        if decision is None:
            decision = PageDecision(
                page_number=page.page_number,
                document_type="photos" if page.is_image_only else "other",
                starts_new_document=False,
                title=_first_line(page.text),
                confidence=0.2,
                reason="Classifier omitted this page; fallback decision inserted.",
            )
        if page.page_number == 1:
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


def _first_line(text: str, limit: int = 90) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:limit]
    return ""

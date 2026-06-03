from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


DOCUMENT_CATEGORIES = {
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

RULE_KEYWORDS = {
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
}

DocumentType = Enum(
    "DocumentType",
    {name: name for name in DOCUMENT_CATEGORIES},
    type=str,
)

DEFAULT_BATCH_SIZE = 5
IMAGE_DETAIL = "high"

SYSTEM_PROMPT = (
    "You are a claim-file document boundary detector and classifier. "
    "Return only structured data."
)

USER_PROMPT = (
    "Classify the attached target page images from one insurance claim file. "
    "Use rolling context only to decide whether the first page continues an "
    "already open document. Classify only the target pages listed in the text "
    "metadata and attached as images. Do not classify pages that appear only "
    "inside rolling_context."
)


class PageDecisionOutput(BaseModel):
    page: int = Field(description="The one-based source PDF page number.")
    document_type: DocumentType = Field(
        description="The configured document category for this page."
    )
    starts_new_document: bool = Field(
        description="True when this page starts a new logical document."
    )
    title: str = Field(description="Short visible or inferred document title.")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence from 0 to 1 for this page decision.",
    )
    reason: str = Field(description="Short reason for the boundary/type decision.")


class BatchClassificationOutput(BaseModel):
    pages: list[PageDecisionOutput]
